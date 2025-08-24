#!/usr/bin/env python3
"""
Phase 2A Execution Script - ML Model Development

This script orchestrates the complete Phase 2A pipeline including:
1. Data loading and preprocessing
2. Feature engineering
3. Baseline model training
4. Advanced model training
5. Model evaluation and comparison
6. Report generation

Usage:
    python scripts/phase_2a_execution.py
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
from data_pipeline.etl import ETLPipeline
from models.feature_engineering import HealthcareFeatureEngineer
from models.baseline_models import HealthcareBaselineModels
from models.advanced_models import AdvancedHealthcareModels
from models.model_evaluation import HealthcareModelEvaluator
from utils.metrics_tracker import MetricsTracker, TimingContext

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase_2a_execution.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class Phase2AExecutor:
    """Main executor for Phase 2A ML model development."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.start_time = datetime.now()
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Initialize components
        self.etl_pipeline = ETLPipeline()
        self.feature_engineer = HealthcareFeatureEngineer()
        self.baseline_models = HealthcareBaselineModels()
        self.advanced_models = AdvancedHealthcareModels()
        self.model_evaluator = HealthcareModelEvaluator()
        
    def run_phase_2a(self) -> Dict[str, Any]:
        """Run complete Phase 2A pipeline."""
        self.logger.info("Starting Phase 2A: ML Model Development")
        self.logger.info(f"Start time: {self.start_time}")
        
        try:
            # Step 1: Data Pipeline Execution
            self.logger.info("📊 Step 1: Data Pipeline Execution")
            etl_results = self._run_data_pipeline()
            
            # Step 2: Feature Engineering
            self.logger.info("🔧 Step 2: Feature Engineering")
            feature_results = self._run_feature_engineering()
            
            # Step 3: Baseline Model Training
            self.logger.info("📈 Step 3: Baseline Model Training")
            baseline_results = self._run_baseline_models()
            
            # Step 4: Advanced Model Training
            self.logger.info("🧠 Step 4: Advanced Model Training")
            advanced_results = self._run_advanced_models()
            
            # Step 5: Model Evaluation and Comparison
            self.logger.info("📊 Step 5: Model Evaluation and Comparison")
            evaluation_results = self._run_model_evaluation()
            
            # Step 6: Generate Reports
            self.logger.info("📋 Step 6: Report Generation")
            report_results = self._generate_reports()
            
            # Compile final results
            self.results = {
                'phase_2a_summary': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                    'status': 'completed',
                    'steps_completed': 6
                },
                'etl_results': etl_results,
                'feature_results': feature_results,
                'baseline_results': baseline_results,
                'advanced_results': advanced_results,
                'evaluation_results': evaluation_results,
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
            
            self.logger.info("Phase 2A completed successfully!")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Phase 2A failed: {str(e)}")
            
            # Save metrics even on failure
            self.metrics_tracker.track_error("Phase_2A_Failure", str(e), "Main_Execution")
            metrics_file = self.metrics_tracker.save_metrics()
            
            self.results['phase_2a_summary'] = {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e),
                'metrics_file': str(metrics_file)
            }
            raise
    
    def _run_data_pipeline(self) -> Dict[str, Any]:
        """Run ETL pipeline."""
        self.logger.info("Running ETL pipeline...")
        
        try:
            with TimingContext(self.metrics_tracker, "ETL_Pipeline"):
                # Run ETL pipeline
                etl_result = self.etl_pipeline.run_pipeline()
                
                # Track data quality metrics
                for table_name, metrics in etl_result.get('quality_metrics', {}).items():
                    self.metrics_tracker.track_data_quality(table_name, metrics)
            
            self.logger.info(f"ETL completed: {etl_result['tables_processed']} tables processed")
            return etl_result
            
        except Exception as e:
            self.metrics_tracker.track_error("ETL_Error", str(e), "ETL_Pipeline")
            self.logger.error(f"ETL pipeline failed: {str(e)}")
            raise
    
    def _run_feature_engineering(self) -> Dict[str, Any]:
        """Run feature engineering."""
        self.logger.info("Running feature engineering...")
        
        try:
            with TimingContext(self.metrics_tracker, "Feature_Engineering"):
                # Load processed data
                data_path = Path("data/processed/parquet")
                
                # Load main tables
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
                    'best_mape': baseline_evaluation[best_model_name]['mape'],
                    'evaluation_results': baseline_evaluation,
                    'forecast_report': self.baseline_models.generate_forecast_report()
                }
            
            self.logger.info(f"Baseline models completed: {baseline_results['models_trained']} models trained")
            self.logger.info(f"Best baseline model: {best_model_name} (MAPE: {baseline_results['best_mape']:.2f}%)")
            
            return baseline_results
            
        except Exception as e:
            self.metrics_tracker.track_error("Baseline_Models_Error", str(e), "Baseline_Models")
            self.logger.error(f"Baseline model training failed: {str(e)}")
            raise
    
    def _run_advanced_models(self) -> Dict[str, Any]:
        """Run advanced model training."""
        self.logger.info("Running advanced model training...")
        
        try:
            # Load engineered features
            feature_df = pd.read_parquet("data/processed/engineered_features.parquet")
            
            # Train advanced models
            advanced_models = self.advanced_models.train_all_advanced_models(feature_df)
            
            # Get evaluation results
            advanced_evaluation = self.advanced_models.evaluation_results
            
            # Get best model
            best_model_name, best_model = self.advanced_models.get_best_advanced_model()
            
            advanced_results = {
                'models_trained': len(advanced_models),
                'best_model': best_model_name,
                'best_mape': advanced_evaluation[best_model_name]['mape'],
                'evaluation_results': advanced_evaluation,
                'advanced_report': self.advanced_models.generate_advanced_model_report()
            }
            
            self.logger.info(f"Advanced models completed: {advanced_results['models_trained']} models trained")
            self.logger.info(f"Best advanced model: {best_model_name} (MAPE: {advanced_results['best_mape']:.2f}%)")
            
            return advanced_results
            
        except Exception as e:
            self.logger.error(f"Advanced model training failed: {str(e)}")
            raise
    
    def _run_model_evaluation(self) -> Dict[str, Any]:
        """Run model evaluation and comparison."""
        self.logger.info("Running model evaluation and comparison...")
        
        try:
            # Load data for evaluation
            encounters_df = pd.read_parquet("data/processed/parquet/encounters.parquet")
            patients_df = pd.read_parquet("data/processed/parquet/patients.parquet")
            conditions_df = pd.read_parquet("data/processed/parquet/conditions.parquet")
            medications_df = pd.read_parquet("data/processed/parquet/medications.parquet")
            
            # Run complete evaluation
            evaluation_results = self.model_evaluator.run_complete_evaluation(
                encounters_df, patients_df, conditions_df, medications_df
            )
            
            self.logger.info("Model evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def _generate_reports(self) -> Dict[str, Any]:
        """Generate comprehensive reports."""
        self.logger.info("Generating reports...")
        
        try:
            report_results = {}
            
            # Save evaluation results
            self.model_evaluator.save_evaluation_results("models/phase_2a_evaluation_results.json")
            report_results['evaluation_results_saved'] = "models/phase_2a_evaluation_results.json"
            
            # Generate visualization report
            self.model_evaluator.generate_visualization_report("models/phase_2a_visualization_report.html")
            report_results['visualization_report_saved'] = "models/phase_2a_visualization_report.html"
            
            # Generate phase summary report
            phase_summary = self._generate_phase_summary()
            report_results['phase_summary'] = phase_summary
            
            # Save phase results
            with open("models/phase_2a_results.json", "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            report_results['phase_results_saved'] = "models/phase_2a_results.json"
            
            self.logger.info("Reports generated successfully")
            return report_results
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise
    
    def _generate_phase_summary(self) -> Dict[str, Any]:
        """Generate phase summary report."""
        evaluation_results = self.results.get('evaluation_results', {})
        report = evaluation_results.get('report', {})
        
        # Get key metrics
        executive_summary = report.get('executive_summary', {})
        overall_performance = executive_summary.get('overall_performance', {})
        
        phase_summary = {
            'phase': '2A - ML Model Development',
            'status': 'completed',
            'target_achieved': overall_performance.get('target_achieved', False),
            'best_model': overall_performance.get('best_model', 'Unknown'),
            'best_mape': overall_performance.get('best_mape', float('inf')),
            'performance_category': overall_performance.get('performance_category', 'Unknown'),
            'deployment_readiness': executive_summary.get('deployment_readiness', {}),
            'recommendations': report.get('recommendations', []),
            'next_steps': report.get('next_steps', []),
            'timestamp': datetime.now().isoformat()
        }
        
        return phase_summary
    
    def print_summary(self):
        """Print phase summary."""
        if not self.results:
            self.logger.warning("No results available. Run phase_2a first.")
            return
        
        summary = self.results.get('phase_2a_summary', {})
        phase_summary = self.results.get('report_results', {}).get('phase_summary', {})
        
        print("\n" + "="*80)
        print("🎯 PHASE 2A SUMMARY - ML MODEL DEVELOPMENT")
        print("="*80)
        
        print(f"📅 Duration: {summary.get('duration_minutes', 0):.1f} minutes")
        print(f"✅ Status: {summary.get('status', 'Unknown')}")
        print(f"📊 Steps Completed: {summary.get('steps_completed', 0)}")
        
        print("\n🏆 PERFORMANCE RESULTS:")
        print(f"   🎯 Target MAPE <8%: {'✅ ACHIEVED' if phase_summary.get('target_achieved') else '❌ NOT ACHIEVED'}")
        print(f"   🏅 Best Model: {phase_summary.get('best_model', 'Unknown')}")
        print(f"   📈 Best MAPE: {phase_summary.get('best_mape', 0):.2f}%")
        print(f"   📊 Performance Category: {phase_summary.get('performance_category', 'Unknown')}")
        
        deployment = phase_summary.get('deployment_readiness', {})
        print(f"\n🚀 DEPLOYMENT READINESS:")
        print(f"   🎯 Ready for Production: {'✅ YES' if deployment.get('ready_for_production') else '❌ NO'}")
        print(f"   🎯 Confidence Level: {deployment.get('confidence_level', 'Unknown')}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in phase_summary.get('recommendations', [])[:3]:  # Show top 3
            print(f"   • {rec}")
        
        print(f"\n🎯 NEXT STEPS:")
        for step in phase_summary.get('next_steps', [])[:3]:  # Show top 3
            print(f"   • {step}")
        
        print("\n" + "="*80)


def main():
    """Main execution function."""
    logger.info("Starting Phase 2A execution")
    
    # Create executor
    executor = Phase2AExecutor()
    
    try:
        # Run Phase 2A
        results = executor.run_phase_2a()
        
        # Print summary
        executor.print_summary()
        
        # Check if we can proceed to Phase 2B
        phase_summary = results.get('report_results', {}).get('phase_summary', {})
        target_achieved = phase_summary.get('target_achieved', False)
        
        if target_achieved:
            logger.info("🎉 Phase 2A completed successfully! Ready to proceed to Phase 2B.")
            return 0
        else:
            logger.warning("⚠️ Phase 2A completed but target not achieved. Review before proceeding to Phase 2B.")
            return 1
            
    except Exception as e:
        logger.error(f"Phase 2A execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
