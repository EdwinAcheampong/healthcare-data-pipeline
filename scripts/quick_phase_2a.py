#!/usr/bin/env python3
"""
Quick Phase 2A Execution - Simplified version for rapid testing.

This script runs a simplified version of Phase 2A that skips slow operations
while maintaining comprehensive metrics tracking.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our modules
from models.feature_engineering import HealthcareFeatureEngineer
from models.baseline_models import HealthcareBaselineModels
from utils.metrics_tracker import MetricsTracker, TimingContext

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quick_phase_2a.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class QuickPhase2AExecutor:
    """Quick executor for Phase 2A ML model development."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.start_time = datetime.now()
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(run_id=f"quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Initialize components
        self.feature_engineer = HealthcareFeatureEngineer()
        self.baseline_models = HealthcareBaselineModels()
        
    def run_quick_phase_2a(self) -> Dict[str, Any]:
        """Run quick Phase 2A pipeline."""
        self.logger.info("🚀 Starting Quick Phase 2A: ML Model Development")
        self.logger.info(f"Start time: {self.start_time}")
        
        try:
            # Step 1: Load existing processed data (skip ETL)
            self.logger.info("📊 Step 1: Loading Processed Data")
            data_results = self._load_processed_data()
            
            # Step 2: Feature Engineering
            self.logger.info("🔧 Step 2: Feature Engineering")
            feature_results = self._run_feature_engineering()
            
            # Step 3: Baseline Model Training
            self.logger.info("📈 Step 3: Baseline Model Training")
            baseline_results = self._run_baseline_models()
            
            # Step 4: Generate Reports
            self.logger.info("📋 Step 4: Report Generation")
            report_results = self._generate_reports()
            
            # Compile final results
            self.results = {
                'phase_2a_summary': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                    'status': 'completed',
                    'steps_completed': 4,
                    'version': 'quick_execution'
                },
                'data_results': data_results,
                'feature_results': feature_results,
                'baseline_results': baseline_results,
                'report_results': report_results
            }
            
            # Save comprehensive metrics
            metrics_file = self.metrics_tracker.save_metrics()
            
            # Generate and log performance report
            performance_report = self.metrics_tracker.generate_performance_report()
            self.logger.info("\n" + performance_report)
            
            # Add metrics to results
            self.results['metrics_file'] = str(metrics_file)
            self.results['performance_summary'] = self.metrics_tracker.calculate_overall_performance()
            
            self.logger.info("🎉 Quick Phase 2A completed successfully!")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Quick Phase 2A failed: {str(e)}")
            
            # Save metrics even on failure
            self.metrics_tracker.track_error("Quick_Phase_2A_Failure", str(e), "Main_Execution")
            metrics_file = self.metrics_tracker.save_metrics()
            
            self.results['phase_2a_summary'] = {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e),
                'metrics_file': str(metrics_file)
            }
            raise
    
    def _load_processed_data(self) -> Dict[str, Any]:
        """Load existing processed data."""
        self.logger.info("Loading existing processed data...")
        
        try:
            with TimingContext(self.metrics_tracker, "Data_Loading"):
                data_path = Path("data/processed/parquet")
                
                if not data_path.exists():
                    raise FileNotFoundError("Processed data not found. Please run full ETL pipeline first.")
                
                # Check for required files
                required_files = ["encounters.parquet", "patients.parquet", "conditions.parquet", "medications.parquet"]
                missing_files = [f for f in required_files if not (data_path / f).exists()]
                
                if missing_files:
                    raise FileNotFoundError(f"Missing required files: {missing_files}")
                
                # Quick data shape check
                encounters_df = pd.read_parquet(data_path / "encounters.parquet")
                patients_df = pd.read_parquet(data_path / "patients.parquet")
                conditions_df = pd.read_parquet(data_path / "conditions.parquet")
                medications_df = pd.read_parquet(data_path / "medications.parquet")
                
                data_results = {
                    'encounters_shape': encounters_df.shape,
                    'patients_shape': patients_df.shape,
                    'conditions_shape': conditions_df.shape,
                    'medications_shape': medications_df.shape,
                    'data_path': str(data_path),
                    'load_method': 'existing_parquet'
                }
                
                self.logger.info(f"Data loaded: {encounters_df.shape[0]} encounters, {patients_df.shape[0]} patients")
                return data_results
                
        except Exception as e:
            self.metrics_tracker.track_error("Data_Loading_Error", str(e), "Data_Loading")
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _run_feature_engineering(self) -> Dict[str, Any]:
        """Run feature engineering."""
        self.logger.info("Running feature engineering...")
        
        try:
            with TimingContext(self.metrics_tracker, "Feature_Engineering"):
                # Load processed data
                data_path = Path("data/processed/parquet")
                
                encounters_df = pd.read_parquet(data_path / "encounters.parquet")
                patients_df = pd.read_parquet(data_path / "patients.parquet")
                conditions_df = pd.read_parquet(data_path / "conditions.parquet")
                medications_df = pd.read_parquet(data_path / "medications.parquet")
                
                self.logger.info(f"Loaded data: {len(encounters_df)} encounters, {len(patients_df)} patients")
                
                # Run feature engineering
                feature_df = self.feature_engineer.engineer_features(
                    encounters_df, patients_df, conditions_df, medications_df
                )
                
                # Save engineered features
                feature_df.to_parquet("data/processed/engineered_features.parquet", index=False)
                
                feature_results = {
                    'total_features': len(self.feature_engineer.feature_names),
                    'feature_names': self.feature_engineer.feature_names,
                    'data_shape': feature_df.shape,
                    'output_path': "data/processed/engineered_features.parquet"
                }
                
                # Track feature engineering results
                self.metrics_tracker.track_feature_engineering(feature_results)
            
            self.logger.info(f"Feature engineering completed: {feature_results['total_features']} features created")
            return feature_results
            
        except Exception as e:
            self.metrics_tracker.track_error("Feature_Engineering_Error", str(e), "Feature_Engineering")
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def _run_baseline_models(self) -> Dict[str, Any]:
        """Run baseline model training."""
        self.logger.info("Running baseline model training...")
        
        try:
            with TimingContext(self.metrics_tracker, "Baseline_Models"):
                # Load engineered features for baseline models
                feature_df = pd.read_parquet("data/processed/engineered_features.parquet")
                
                # Train baseline models
                baseline_models = self.baseline_models.train_all_baseline_models(feature_df)
                
                # Get evaluation results
                baseline_evaluation = self.baseline_models.evaluation_results
                
                # Track model performance
                for model_name, metrics in baseline_evaluation.items():
                    self.metrics_tracker.track_model_performance(f"baseline_{model_name}", metrics)
                
                # Get best model
                best_model_name, best_model = self.baseline_models.get_best_model()
                
                baseline_results = {
                    'models_trained': len(baseline_models),
                    'best_model': best_model_name,
                    'best_mape': baseline_evaluation[best_model_name]['mape'] if best_model_name else float('inf'),
                    'evaluation_results': baseline_evaluation,
                    'forecast_report': self.baseline_models.generate_forecast_report()
                }
            
            self.logger.info(f"Baseline models completed: {baseline_results['models_trained']} models trained")
            if best_model_name:
                self.logger.info(f"Best baseline model: {best_model_name} (MAPE: {baseline_results['best_mape']:.2f}%)")
            
            return baseline_results
            
        except Exception as e:
            self.metrics_tracker.track_error("Baseline_Models_Error", str(e), "Baseline_Models")
            self.logger.error(f"Baseline model training failed: {str(e)}")
            raise
    
    def _generate_reports(self) -> Dict[str, Any]:
        """Generate comprehensive reports."""
        self.logger.info("Generating reports...")
        
        try:
            with TimingContext(self.metrics_tracker, "Report_Generation"):
                report_results = {}
                
                # Save quick results summary
                quick_results_file = f"models/quick_phase_2a_results_{self.metrics_tracker.run_id}.json"
                with open(quick_results_file, "w") as f:
                    json.dump(self.results, f, indent=2, default=str)
                report_results['quick_results_saved'] = quick_results_file
                
                # Generate quick summary report
                summary = self._generate_quick_summary()
                report_results['quick_summary'] = summary
                
                self.logger.info("Reports generated successfully")
                return report_results
                
        except Exception as e:
            self.metrics_tracker.track_error("Report_Generation_Error", str(e), "Report_Generation")
            self.logger.error(f"Report generation failed: {str(e)}")
            raise
    
    def _generate_quick_summary(self) -> Dict[str, Any]:
        """Generate quick phase summary report."""
        baseline_results = self.results.get('baseline_results', {})
        feature_results = self.results.get('feature_results', {})
        
        summary = {
            'phase': '2A - Quick ML Model Development',
            'status': 'completed',
            'target_achieved': baseline_results.get('best_mape', float('inf')) < 8.0,
            'best_model': baseline_results.get('best_model', 'Unknown'),
            'best_mape': baseline_results.get('best_mape', float('inf')),
            'models_trained': baseline_results.get('models_trained', 0),
            'features_created': feature_results.get('total_features', 0),
            'execution_method': 'quick_pipeline',
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def print_summary(self):
        """Print phase summary."""
        if not self.results:
            self.logger.warning("No results available. Run quick_phase_2a first.")
            return
        
        summary = self.results.get('phase_2a_summary', {})
        quick_summary = self.results.get('report_results', {}).get('quick_summary', {})
        
        print("\n" + "="*80)
        print("🚀 QUICK PHASE 2A SUMMARY - ML MODEL DEVELOPMENT")
        print("="*80)
        
        print(f"📅 Duration: {summary.get('duration_minutes', 0):.1f} minutes")
        print(f"✅ Status: {summary.get('status', 'Unknown')}")
        print(f"📊 Steps Completed: {summary.get('steps_completed', 0)}")
        print(f"🔄 Execution Method: {summary.get('version', 'Unknown')}")
        
        print("\n🏆 PERFORMANCE RESULTS:")
        print(f"   🎯 Target MAPE <8%: {'✅ ACHIEVED' if quick_summary.get('target_achieved') else '❌ NOT ACHIEVED'}")
        print(f"   🏅 Best Model: {quick_summary.get('best_model', 'Unknown')}")
        print(f"   📈 Best MAPE: {quick_summary.get('best_mape', 0):.2f}%")
        print(f"   🤖 Models Trained: {quick_summary.get('models_trained', 0)}")
        print(f"   🔧 Features Created: {quick_summary.get('features_created', 0)}")
        
        performance_summary = self.results.get('performance_summary', {})
        print(f"\n📊 RESOURCE USAGE:")
        print(f"   ⏱️  Total Time: {performance_summary.get('total_execution_time_minutes', 0):.2f} minutes")
        print(f"   ✅ Success Rate: {performance_summary.get('success_rate_percentage', 0):.1f}%")
        print(f"   🚀 Optimizations: {performance_summary.get('optimizations_applied', 0)}")
        print(f"   ⚠️  Errors: {performance_summary.get('errors_encountered', 0)}")
        
        metrics_file = self.results.get('metrics_file', 'Unknown')
        print(f"\n💾 Full metrics saved to: {metrics_file}")
        
        print("\n" + "="*80)


def main():
    """Main execution function."""
    logger.info("Starting Quick Phase 2A execution")
    
    # Create executor
    executor = QuickPhase2AExecutor()
    
    try:
        # Run Quick Phase 2A
        results = executor.run_quick_phase_2a()
        
        # Print summary
        executor.print_summary()
        
        # Check if we achieved target
        quick_summary = results.get('report_results', {}).get('quick_summary', {})
        target_achieved = quick_summary.get('target_achieved', False)
        
        if target_achieved:
            logger.info("🎉 Quick Phase 2A completed successfully! Target achieved.")
            return 0
        else:
            logger.warning("⚠️ Quick Phase 2A completed but target not achieved.")
            return 1
            
    except Exception as e:
        logger.error(f"Quick Phase 2A execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
