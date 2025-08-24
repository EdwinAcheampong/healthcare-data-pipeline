#!/usr/bin/env python3
"""
Test Script for Phase 2A Implementation

This script tests each component of Phase 2A to ensure everything works correctly:
1. Feature Engineering
2. Baseline Models
3. Advanced Models
4. Model Evaluation
5. Complete Pipeline

Usage:
    python scripts/test_phase_2a.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our modules
from models.feature_engineering import HealthcareFeatureEngineer
from models.baseline_models import HealthcareBaselineModels
from models.advanced_models import AdvancedHealthcareModels
from models.model_evaluation import HealthcareModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class Phase2ATester:
    """Test suite for Phase 2A components."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.start_time = datetime.now()
        
        # Initialize components
        self.feature_engineer = HealthcareFeatureEngineer()
        self.baseline_models = HealthcareBaselineModels()
        self.advanced_models = AdvancedHealthcareModels()
        self.model_evaluator = HealthcareModelEvaluator()
        
    def create_test_data(self) -> Dict[str, pd.DataFrame]:
        """Create synthetic test data for validation."""
        self.logger.info("Creating synthetic test data...")
        
        # Create test encounters
        np.random.seed(42)
        n_encounters = 1000
        n_patients = 200
        
        # Generate timestamps
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        timestamps = pd.date_range(start_date, end_date, periods=n_encounters)
        
        # Create encounters data
        encounters_data = {
            'ENCOUNTER': [f'enc_{i}' for i in range(n_encounters)],
            'PATIENT': [f'pat_{np.random.randint(0, n_patients)}' for _ in range(n_encounters)],
            'START': timestamps,
            'STOP': timestamps + timedelta(hours=np.random.randint(1, 8)),
            'ENCOUNTERCLASS': np.random.choice(['ambulatory', 'emergency', 'inpatient'], n_encounters),
            'PROVIDER': [f'prov_{np.random.randint(0, 50)}' for _ in range(n_encounters)],
            'ORGANIZATION': [f'org_{np.random.randint(0, 10)}' for _ in range(n_encounters)]
        }
        encounters_df = pd.DataFrame(encounters_data)
        
        # Create patients data
        patients_data = {
            'Id': [f'pat_{i}' for i in range(n_patients)],
            'BIRTHDATE': [start_date - timedelta(days=np.random.randint(365*18, 365*80)) for _ in range(n_patients)],
            'GENDER': np.random.choice(['M', 'F'], n_patients),
            'RACE': np.random.choice(['white', 'black', 'asian', 'hispanic'], n_patients),
            'ETHNICITY': np.random.choice(['nonhispanic', 'hispanic'], n_patients),
            'PASSPORT': [f'pass_{i}' if np.random.random() > 0.3 else None for i in range(n_patients)]
        }
        patients_df = pd.DataFrame(patients_data)
        
        # Create conditions data
        conditions_data = {
            'PATIENT': [f'pat_{np.random.randint(0, n_patients)}' for _ in range(n_encounters * 2)],
            'ENCOUNTER': [f'enc_{np.random.randint(0, n_encounters)}' for _ in range(n_encounters * 2)],
            'START': [timestamps[np.random.randint(0, len(timestamps))] for _ in range(n_encounters * 2)],
            'DESCRIPTION': np.random.choice([
                'COVID-19', 'Diabetes mellitus', 'Hypertension', 'Asthma',
                'Heart disease', 'Acute respiratory infection', 'Chronic pain'
            ], n_encounters * 2)
        }
        conditions_df = pd.DataFrame(conditions_data)
        
        # Create medications data
        medications_data = {
            'PATIENT': [f'pat_{np.random.randint(0, n_patients)}' for _ in range(n_encounters * 3)],
            'ENCOUNTER': [f'enc_{np.random.randint(0, n_encounters)}' for _ in range(n_encounters * 3)],
            'START': [timestamps[np.random.randint(0, len(timestamps))] for _ in range(n_encounters * 3)],
            'DESCRIPTION': np.random.choice([
                'Acetaminophen', 'Ibuprofen', 'Amoxicillin', 'Insulin',
                'Morphine', 'Aspirin', 'Azithromycin'
            ], n_encounters * 3)
        }
        medications_df = pd.DataFrame(medications_data)
        
        test_data = {
            'encounters': encounters_df,
            'patients': patients_df,
            'conditions': conditions_df,
            'medications': medications_df
        }
        
        self.logger.info(f"Created test data: {len(encounters_df)} encounters, {len(patients_df)} patients")
        return test_data
    
    def test_feature_engineering(self, test_data: Dict[str, pd.DataFrame]) -> bool:
        """Test feature engineering component."""
        self.logger.info("Testing feature engineering...")
        
        try:
            # Run feature engineering
            feature_df = self.feature_engineer.engineer_features(
                test_data['encounters'],
                test_data['patients'],
                test_data['conditions'],
                test_data['medications']
            )
            
            # Validate results
            assert len(feature_df) > 0, "Feature dataframe is empty"
            assert len(self.feature_engineer.feature_names) > 0, "No features created"
            assert 'encounters_last_24h' in feature_df.columns, "Missing workload features"
            assert 'is_covid_related' in feature_df.columns, "Missing condition features"
            
            self.test_results['feature_engineering'] = {
                'status': 'passed',
                'features_created': len(self.feature_engineer.feature_names),
                'data_shape': feature_df.shape,
                'feature_names': self.feature_engineer.feature_names[:5]  # First 5 features
            }
            
            self.logger.info(f"✅ Feature engineering test passed: {len(self.feature_engineer.feature_names)} features created")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering test failed: {str(e)}")
            self.test_results['feature_engineering'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_baseline_models(self, test_data: Dict[str, pd.DataFrame]) -> bool:
        """Test baseline models component."""
        self.logger.info("Testing baseline models...")
        
        try:
            # First run feature engineering to get the target column
            feature_df = self.feature_engineer.engineer_features(
                test_data['encounters'],
                test_data['patients'],
                test_data['conditions'],
                test_data['medications']
            )
            
            # Run baseline model training with feature-engineered data
            baseline_models = self.baseline_models.train_all_baseline_models(feature_df)
            
            # Validate results
            assert len(baseline_models) > 0, "No baseline models trained"
            assert len(self.baseline_models.evaluation_results) > 0, "No evaluation results"
            
            # Get best model
            best_model_name, best_model = self.baseline_models.get_best_model()
            assert best_model_name is not None, "No best model found"
            
            self.test_results['baseline_models'] = {
                'status': 'passed',
                'models_trained': len(baseline_models),
                'best_model': best_model_name,
                'best_mape': self.baseline_models.evaluation_results[best_model_name]['mape'],
                'evaluation_results': self.baseline_models.evaluation_results
            }
            
            self.logger.info(f"✅ Baseline models test passed: {len(baseline_models)} models trained")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Baseline models test failed: {str(e)}")
            self.test_results['baseline_models'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_advanced_models(self, test_data: Dict[str, pd.DataFrame]) -> bool:
        """Test advanced models component."""
        self.logger.info("Testing advanced models...")
        
        try:
            # First run feature engineering to get features
            feature_df = self.feature_engineer.engineer_features(
                test_data['encounters'],
                test_data['patients'],
                test_data['conditions'],
                test_data['medications']
            )
            
            # Run advanced model training
            advanced_models = self.advanced_models.train_all_advanced_models(feature_df)
            
            # Validate results
            assert len(advanced_models) > 0, "No advanced models trained"
            assert len(self.advanced_models.evaluation_results) > 0, "No evaluation results"
            
            # Get best model
            best_model_name, best_model = self.advanced_models.get_best_advanced_model()
            assert best_model_name is not None, "No best model found"
            
            self.test_results['advanced_models'] = {
                'status': 'passed',
                'models_trained': len(advanced_models),
                'best_model': best_model_name,
                'best_mape': self.advanced_models.evaluation_results[best_model_name]['mape'],
                'evaluation_results': self.advanced_models.evaluation_results
            }
            
            self.logger.info(f"✅ Advanced models test passed: {len(advanced_models)} models trained")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Advanced models test failed: {str(e)}")
            self.test_results['advanced_models'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_model_evaluation(self, test_data: Dict[str, pd.DataFrame]) -> bool:
        """Test model evaluation component."""
        self.logger.info("Testing model evaluation...")
        
        try:
            # Run complete evaluation
            evaluation_results = self.model_evaluator.run_complete_evaluation(
                test_data['encounters'],
                test_data['patients'],
                test_data['conditions'],
                test_data['medications']
            )
            
            # Validate results
            assert 'baseline' in evaluation_results, "Missing baseline results"
            assert 'advanced' in evaluation_results, "Missing advanced results"
            assert 'comparison' in evaluation_results, "Missing comparison results"
            assert 'report' in evaluation_results, "Missing report"
            
            # Check comparison summary
            comparison = evaluation_results['comparison']
            assert 'summary' in comparison, "Missing comparison summary"
            assert 'overall_best' in comparison['summary'], "Missing overall best model"
            
            self.test_results['model_evaluation'] = {
                'status': 'passed',
                'baseline_models': len(evaluation_results['baseline']),
                'advanced_models': len(evaluation_results['advanced']),
                'overall_best': comparison['summary']['overall_best'],
                'overall_best_mape': comparison['summary']['overall_best_mape']
            }
            
            self.logger.info(f"✅ Model evaluation test passed: {comparison['summary']['overall_best']} is best model")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation test failed: {str(e)}")
            self.test_results['model_evaluation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_complete_pipeline(self, test_data: Dict[str, pd.DataFrame]) -> bool:
        """Test complete Phase 2A pipeline."""
        self.logger.info("Testing complete Phase 2A pipeline...")
        
        try:
            # Test all components in sequence
            feature_success = self.test_feature_engineering(test_data)
            baseline_success = self.test_baseline_models(test_data)
            advanced_success = self.test_advanced_models(test_data)
            evaluation_success = self.test_model_evaluation(test_data)
            
            # Check if all tests passed
            all_passed = all([feature_success, baseline_success, advanced_success, evaluation_success])
            
            self.test_results['complete_pipeline'] = {
                'status': 'passed' if all_passed else 'failed',
                'feature_engineering': feature_success,
                'baseline_models': baseline_success,
                'advanced_models': advanced_success,
                'model_evaluation': evaluation_success,
                'all_tests_passed': all_passed
            }
            
            if all_passed:
                self.logger.info("✅ Complete Phase 2A pipeline test passed!")
            else:
                self.logger.error("❌ Complete Phase 2A pipeline test failed!")
            
            return all_passed
            
        except Exception as e:
            self.logger.error(f"❌ Complete pipeline test failed: {str(e)}")
            self.test_results['complete_pipeline'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 2A tests."""
        self.logger.info("🚀 Starting Phase 2A test suite")
        self.logger.info(f"Start time: {self.start_time}")
        
        # Create test data
        test_data = self.create_test_data()
        
        # Run complete pipeline test
        pipeline_success = self.test_complete_pipeline(test_data)
        
        # Compile results
        results = {
            'test_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                'overall_status': 'passed' if pipeline_success else 'failed',
                'tests_passed': sum(1 for result in self.test_results.values() if result.get('status') == 'passed'),
                'total_tests': len(self.test_results)
            },
            'test_results': self.test_results
        }
        
        # Save results
        with open("models/phase_2a_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info("Test suite completed")
        return results
    
    def print_test_summary(self):
        """Print test summary."""
        if not self.test_results:
            self.logger.warning("No test results available. Run tests first.")
            return
        
        print("\n" + "="*80)
        print("🧪 PHASE 2A TEST SUMMARY")
        print("="*80)
        
        # Count passed/failed tests
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'passed')
        failed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'failed')
        total_tests = len(self.test_results)
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
        print(f"⏱️ Duration: {(datetime.now() - self.start_time).total_seconds() / 60:.1f} minutes")
        
        print(f"\n📋 Component Tests:")
        for component, result in self.test_results.items():
            status_icon = "✅" if result.get('status') == 'passed' else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'passed':
                if 'features_created' in result:
                    print(f"      📈 Features created: {result['features_created']}")
                if 'models_trained' in result:
                    print(f"      🤖 Models trained: {result['models_trained']}")
                if 'best_model' in result:
                    print(f"      🏆 Best model: {result['best_model']}")
                if 'best_mape' in result:
                    print(f"      📊 Best MAPE: {result['best_mape']:.2f}%")
        
        # Overall assessment
        if passed_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! Phase 2A is ready for execution.")
        else:
            print(f"\n⚠️ {failed_tests} test(s) failed. Review before proceeding.")
        
        print("\n" + "="*80)


def main():
    """Main test execution function."""
    logger.info("Starting Phase 2A test suite")
    
    # Create tester
    tester = Phase2ATester()
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        # Print summary
        tester.print_test_summary()
        
        # Return exit code
        overall_status = results['test_summary']['overall_status']
        return 0 if overall_status == 'passed' else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
