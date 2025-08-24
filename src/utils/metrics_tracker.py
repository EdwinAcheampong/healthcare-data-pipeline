"""
Comprehensive Metrics Tracking System for Healthcare Data Pipeline.

This module tracks performance metrics, execution times, model results,
and system improvements across different runs and configurations.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import time
import psutil
import os


class MetricsTracker:
    """Comprehensive metrics tracking for the healthcare data pipeline."""
    
    def __init__(self, run_id: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create metrics directory
        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Initialize tracking data
        self.metrics = {
            'run_info': {
                'run_id': self.run_id,
                'start_time': datetime.now().isoformat(),
                'git_commit': self._get_git_commit(),
                'python_version': self._get_python_version(),
                'system_info': self._get_system_info()
            },
            'execution_times': {},
            'model_performance': {},
            'data_quality': {},
            'feature_engineering': {},
            'resource_usage': {},
            'errors_and_warnings': [],
            'optimizations_applied': [],
            'improvements': {}
        }
        
        # Start resource monitoring
        self.start_time = time.time()
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'platform': os.name,
            'architecture': psutil.cpu_freq().current if psutil.cpu_freq() else "unknown"
        }
    
    def start_phase(self, phase_name: str):
        """Start tracking a phase."""
        self.metrics['execution_times'][phase_name] = {
            'start_time': datetime.now().isoformat(),
            'start_timestamp': time.time()
        }
        self.logger.info(f"📊 Starting metrics tracking for {phase_name}")
    
    def end_phase(self, phase_name: str, success: bool = True, error: str = None):
        """End tracking a phase."""
        if phase_name in self.metrics['execution_times']:
            start_time = self.metrics['execution_times'][phase_name]['start_timestamp']
            duration = time.time() - start_time
            
            self.metrics['execution_times'][phase_name].update({
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'success': success,
                'error': error
            })
            
            # Track resource usage for this phase
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = self.process.cpu_percent()
            
            self.metrics['resource_usage'][phase_name] = {
                'memory_mb': current_memory,
                'memory_increase_mb': current_memory - self.initial_memory,
                'cpu_percent': cpu_percent
            }
            
            self.logger.info(f"📊 Completed {phase_name}: {duration:.2f}s, "
                           f"Memory: {current_memory:.1f}MB, CPU: {cpu_percent:.1f}%")
    
    def track_data_quality(self, table_name: str, metrics: Dict[str, Any]):
        """Track data quality metrics."""
        self.metrics['data_quality'][table_name] = {
            **metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def track_feature_engineering(self, feature_results: Dict[str, Any]):
        """Track feature engineering results."""
        self.metrics['feature_engineering'] = {
            **feature_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"📊 Feature Engineering: {feature_results.get('total_features', 0)} features created")
    
    def track_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Track model performance metrics."""
        self.metrics['model_performance'][model_name] = {
            **metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        mape = metrics.get('mape', float('inf'))
        self.logger.info(f"📊 Model {model_name}: MAPE = {mape:.2f}%")
    
    def track_optimization(self, optimization_name: str, description: str, 
                          before_metric: float, after_metric: float, 
                          metric_name: str = "execution_time"):
        """Track optimization impact."""
        improvement = {
            'optimization': optimization_name,
            'description': description,
            'metric_name': metric_name,
            'before': before_metric,
            'after': after_metric,
            'improvement_absolute': before_metric - after_metric,
            'improvement_percentage': ((before_metric - after_metric) / before_metric * 100) if before_metric > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics['optimizations_applied'].append(improvement)
        self.logger.info(f"📊 Optimization {optimization_name}: {improvement['improvement_percentage']:.1f}% improvement")
    
    def track_error(self, error_type: str, error_message: str, phase: str = None):
        """Track errors and warnings."""
        error_info = {
            'type': error_type,
            'message': error_message,
            'phase': phase,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics['errors_and_warnings'].append(error_info)
        self.logger.warning(f"📊 Tracked error in {phase}: {error_type}")
    
    def calculate_overall_performance(self) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        total_duration = time.time() - self.start_time
        
        # Find best model
        best_model = None
        best_mape = float('inf')
        
        for model_name, metrics in self.metrics['model_performance'].items():
            if metrics.get('mape', float('inf')) < best_mape:
                best_mape = metrics['mape']
                best_model = model_name
        
        # Calculate success rate
        total_phases = len(self.metrics['execution_times'])
        successful_phases = sum(1 for phase in self.metrics['execution_times'].values() 
                               if phase.get('success', False))
        success_rate = (successful_phases / total_phases * 100) if total_phases > 0 else 0
        
        overall_performance = {
            'total_execution_time_minutes': total_duration / 60,
            'best_model': best_model,
            'best_mape': best_mape,
            'target_achieved': best_mape < 8.0,  # Target MAPE < 8%
            'success_rate_percentage': success_rate,
            'total_features_created': self.metrics['feature_engineering'].get('total_features', 0),
            'models_trained': len(self.metrics['model_performance']),
            'optimizations_applied': len(self.metrics['optimizations_applied']),
            'errors_encountered': len(self.metrics['errors_and_warnings'])
        }
        
        self.metrics['improvements'] = overall_performance
        return overall_performance
    
    def save_metrics(self, filename: str = None):
        """Save all metrics to a JSON file."""
        if filename is None:
            filename = f"metrics_{self.run_id}.json"
        
        filepath = self.metrics_dir / filename
        
        # Add final calculations
        self.metrics['run_info']['end_time'] = datetime.now().isoformat()
        self.calculate_overall_performance()
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        self.logger.info(f"📊 Metrics saved to {filepath}")
        return filepath
    
    def generate_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        overall = self.calculate_overall_performance()
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    HEALTHCARE DATA PIPELINE - PERFORMANCE REPORT                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝

🏃 RUN INFORMATION
├─ Run ID: {self.run_id}
├─ Start Time: {self.metrics['run_info']['start_time']}
├─ End Time: {self.metrics['run_info']['end_time']}
├─ Total Duration: {overall['total_execution_time_minutes']:.2f} minutes
└─ Git Commit: {self.metrics['run_info']['git_commit'][:8]}

🎯 OVERALL PERFORMANCE
├─ Target MAPE <8%: {'✅ ACHIEVED' if overall['target_achieved'] else '❌ NOT ACHIEVED'}
├─ Best Model: {overall['best_model'] or 'None'}
├─ Best MAPE: {overall['best_mape']:.2f}%
├─ Success Rate: {overall['success_rate_percentage']:.1f}%
└─ Models Trained: {overall['models_trained']}

📊 FEATURE ENGINEERING
├─ Features Created: {overall['total_features_created']}
├─ Data Shape: {self.metrics['feature_engineering'].get('data_shape', 'Unknown')}
└─ Output Path: {self.metrics['feature_engineering'].get('output_path', 'Unknown')}

⏱️  EXECUTION TIMES
"""
        
        for phase_name, phase_data in self.metrics['execution_times'].items():
            status = "✅" if phase_data.get('success', False) else "❌"
            duration = phase_data.get('duration_minutes', 0)
            report += f"├─ {phase_name}: {status} {duration:.2f} min\n"
        
        if self.metrics['model_performance']:
            report += "\n🤖 MODEL PERFORMANCE\n"
            for model_name, metrics in self.metrics['model_performance'].items():
                mape = metrics.get('mape', float('inf'))
                r2 = metrics.get('r2', 0)
                report += f"├─ {model_name}: MAPE={mape:.2f}%, R²={r2:.3f}\n"
        
        if self.metrics['optimizations_applied']:
            report += "\n🚀 OPTIMIZATIONS APPLIED\n"
            for opt in self.metrics['optimizations_applied']:
                report += f"├─ {opt['optimization']}: {opt['improvement_percentage']:.1f}% improvement\n"
        
        if self.metrics['errors_and_warnings']:
            report += "\n⚠️  ERRORS & WARNINGS\n"
            for error in self.metrics['errors_and_warnings']:
                report += f"├─ {error['phase']}: {error['type']}\n"
        
        report += f"\n💾 Full metrics saved to: metrics/{self.run_id}.json\n"
        
        return report
    
    def compare_with_previous_run(self, previous_metrics_file: str) -> Dict[str, Any]:
        """Compare current run with a previous run."""
        try:
            with open(previous_metrics_file, 'r') as f:
                previous_metrics = json.load(f)
            
            current_perf = self.calculate_overall_performance()
            prev_perf = previous_metrics.get('improvements', {})
            
            comparison = {
                'execution_time_change': {
                    'current': current_perf['total_execution_time_minutes'],
                    'previous': prev_perf.get('total_execution_time_minutes', 0),
                    'improvement_minutes': prev_perf.get('total_execution_time_minutes', 0) - current_perf['total_execution_time_minutes']
                },
                'model_performance_change': {
                    'current_mape': current_perf['best_mape'],
                    'previous_mape': prev_perf.get('best_mape', float('inf')),
                    'mape_improvement': prev_perf.get('best_mape', float('inf')) - current_perf['best_mape']
                },
                'feature_count_change': {
                    'current': current_perf['total_features_created'],
                    'previous': prev_perf.get('total_features_created', 0),
                    'change': current_perf['total_features_created'] - prev_perf.get('total_features_created', 0)
                }
            }
            
            self.metrics['comparison'] = comparison
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing with previous run: {str(e)}")
            return {}


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, tracker: MetricsTracker, operation_name: str):
        self.tracker = tracker
        self.operation_name = operation_name
    
    def __enter__(self):
        self.tracker.start_phase(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self.tracker.end_phase(self.operation_name, success, error)
